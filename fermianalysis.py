from fermipy.gtanalysis import GTAnalysis
import yaml
#!ls *PH*.fits > PH.txt
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
gta = GTAnalysis('config.yaml')
gta.setup()
tsmap_prefit = gta.tsmap(prefix='TSmap_prefit',make_plots=True,write_fits=True,write_npy=True)
resid_prefit = gta.residmap('RSDmap_prefit',model={'SpatialModel' : 'PointSource', 'Index' : 2},write_fits=True,write_npy=True,make_plots=True)
sed_prefit = gta.sed(gta.roi.sources[0].name, bin_index=2.2, outfile='sed_prefit.fits', loge_bins=None,write_npy=True,write_fits=True,make_plots=True)
binned_sed_prefit = gta.sed(gta.roi.sources[0].name, bin_index=2.2, outfile='binned_sed_prefit.fits',loge_bins=[3.0,3.6,4.2,4.8,5.4,6.0],write_npy=True,write_fits=True,make_plots=True)
gta.free_sources(free=True)
gta.optimize()
gta.fit()
tsmap = gta.tsmap(prefix='TSmap',make_plots=True,write_fits=True,write_npy=True)
resid = gta.residmap('RSDmap',model={'SpatialModel' : 'PointSource', 'Index' : 2},write_fits=True,write_npy=True,make_plots=True)
sed = gta.sed(gta.roi.sources[0].name, bin_index=2.2, outfile='sed.fits', loge_bins=None,write_npy=True,write_fits=True,make_plots=True)
binned_sed = gta.sed(gta.roi.sources[0].name, bin_index=2.2, outfile='binned_sed.fits',loge_bins=[3.0,3.6,4.2,4.8,5.4,6.0],write_npy=True,write_fits=True,make_plots=True)
